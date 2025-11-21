# React Native Development Guidelines

This document provides mandatory standards for building cross-platform mobile applications with React Native.

---

**Agent Profile**: The React Native Expert
**Role**: Senior Mobile Developer & Cross-Platform Architect
**Objective**: Generate performant, maintainable React Native applications that provide native-quality experiences on iOS and Android.
**Tools**: React Native 0.73+, Expo, TypeScript, React Navigation, Reanimated.

---

## 1. Core Philosophies: NATIVE-FIRST

- **N**ative feel: Apps should feel native on each platform
- **A**synchronous: Non-blocking operations for smooth UI
- **T**yped: TypeScript for reliability
- **I**solated: Component-based architecture
- **V**erified: Tested on real devices
- **E**fficient: Optimized for mobile constraints

---

## 2. Project Structure (MANDATORY)

### A. Directory Layout

```
src/
├── app/                      # App entry and navigation
│   ├── App.tsx
│   ├── navigation/
│   │   ├── index.tsx
│   │   ├── MainNavigator.tsx
│   │   ├── AuthNavigator.tsx
│   │   └── types.ts
│   └── providers/
│       └── AppProviders.tsx
├── components/               # Reusable components
│   ├── ui/                   # Basic UI components
│   │   ├── Button.tsx
│   │   ├── Input.tsx
│   │   └── Card.tsx
│   ├── forms/
│   └── shared/
├── screens/                  # Screen components
│   ├── auth/
│   │   ├── LoginScreen.tsx
│   │   └── RegisterScreen.tsx
│   ├── home/
│   │   └── HomeScreen.tsx
│   └── profile/
│       └── ProfileScreen.tsx
├── features/                 # Feature modules
│   ├── auth/
│   │   ├── hooks/
│   │   ├── services/
│   │   └── store/
│   └── orders/
├── hooks/                    # Global custom hooks
├── services/                 # API and external services
├── store/                    # Global state management
├── theme/                    # Styling and theming
│   ├── colors.ts
│   ├── spacing.ts
│   ├── typography.ts
│   └── index.ts
├── utils/                    # Utility functions
├── types/                    # TypeScript types
└── constants/                # App constants
```

---

## 3. Component Patterns (MANDATORY)

### A. Functional Components

```tsx
// components/ui/Button.tsx
import React from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
  ViewStyle,
  TextStyle,
} from 'react-native';

interface ButtonProps {
  title: string;
  onPress: () => void;
  variant?: 'primary' | 'secondary' | 'outline';
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  loading?: boolean;
  style?: ViewStyle;
  textStyle?: TextStyle;
}

export function Button({
  title,
  onPress,
  variant = 'primary',
  size = 'medium',
  disabled = false,
  loading = false,
  style,
  textStyle,
}: ButtonProps) {
  const isDisabled = disabled || loading;

  return (
    <TouchableOpacity
      style={[
        styles.base,
        styles[variant],
        styles[size],
        isDisabled && styles.disabled,
        style,
      ]}
      onPress={onPress}
      disabled={isDisabled}
      activeOpacity={0.7}
    >
      {loading ? (
        <ActivityIndicator color={variant === 'primary' ? '#fff' : '#007AFF'} />
      ) : (
        <Text style={[styles.text, styles[`${variant}Text`], textStyle]}>
          {title}
        </Text>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  base: {
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 8,
  },
  primary: {
    backgroundColor: '#007AFF',
  },
  secondary: {
    backgroundColor: '#E5E5EA',
  },
  outline: {
    backgroundColor: 'transparent',
    borderWidth: 1,
    borderColor: '#007AFF',
  },
  small: {
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  medium: {
    paddingVertical: 12,
    paddingHorizontal: 24,
  },
  large: {
    paddingVertical: 16,
    paddingHorizontal: 32,
  },
  disabled: {
    opacity: 0.5,
  },
  text: {
    fontWeight: '600',
  },
  primaryText: {
    color: '#fff',
  },
  secondaryText: {
    color: '#000',
  },
  outlineText: {
    color: '#007AFF',
  },
});
```

### B. Screen Components

```tsx
// screens/home/HomeScreen.tsx
import React, { useCallback } from 'react';
import { View, FlatList, StyleSheet, RefreshControl } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { useOrders } from '@/features/orders/hooks/useOrders';
import { OrderCard } from '@/features/orders/components/OrderCard';
import { EmptyState } from '@/components/shared/EmptyState';
import { LoadingState } from '@/components/shared/LoadingState';
import { ErrorState } from '@/components/shared/ErrorState';
import type { HomeScreenNavigationProp } from '@/app/navigation/types';

export function HomeScreen() {
  const navigation = useNavigation<HomeScreenNavigationProp>();
  const insets = useSafeAreaInsets();

  const {
    orders,
    isLoading,
    isRefreshing,
    error,
    refetch,
    fetchMore,
    hasMore,
  } = useOrders();

  const handleOrderPress = useCallback((orderId: string) => {
    navigation.navigate('OrderDetail', { orderId });
  }, [navigation]);

  const handleEndReached = useCallback(() => {
    if (hasMore && !isLoading) {
      fetchMore();
    }
  }, [hasMore, isLoading, fetchMore]);

  if (isLoading && !orders.length) {
    return <LoadingState />;
  }

  if (error && !orders.length) {
    return <ErrorState message={error.message} onRetry={refetch} />;
  }

  return (
    <View style={[styles.container, { paddingTop: insets.top }]}>
      <FlatList
        data={orders}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <OrderCard order={item} onPress={() => handleOrderPress(item.id)} />
        )}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl refreshing={isRefreshing} onRefresh={refetch} />
        }
        onEndReached={handleEndReached}
        onEndReachedThreshold={0.5}
        ListEmptyComponent={<EmptyState message="No orders yet" />}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F2F2F7',
  },
  listContent: {
    padding: 16,
  },
});
```

---

## 4. Navigation (MANDATORY)

### A. Navigation Setup

```tsx
// app/navigation/index.tsx
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

import { useAuth } from '@/features/auth/hooks/useAuth';
import { AuthNavigator } from './AuthNavigator';
import { MainNavigator } from './MainNavigator';
import { linking } from './linking';

const Stack = createNativeStackNavigator();

export function RootNavigator() {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return <SplashScreen />;
  }

  return (
    <NavigationContainer linking={linking}>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {isAuthenticated ? (
          <Stack.Screen name="Main" component={MainNavigator} />
        ) : (
          <Stack.Screen name="Auth" component={AuthNavigator} />
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}
```

### B. Type-Safe Navigation

```tsx
// app/navigation/types.ts
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import type { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import type { CompositeNavigationProp, RouteProp } from '@react-navigation/native';

// Root stack
export type RootStackParamList = {
  Auth: undefined;
  Main: undefined;
};

// Auth stack
export type AuthStackParamList = {
  Login: undefined;
  Register: undefined;
  ForgotPassword: { email?: string };
};

// Main tab navigator
export type MainTabParamList = {
  Home: undefined;
  Search: undefined;
  Cart: undefined;
  Profile: undefined;
};

// Home stack
export type HomeStackParamList = {
  HomeMain: undefined;
  OrderDetail: { orderId: string };
  ProductDetail: { productId: string };
};

// Navigation prop types
export type HomeScreenNavigationProp = CompositeNavigationProp<
  NativeStackNavigationProp<HomeStackParamList, 'HomeMain'>,
  BottomTabNavigationProp<MainTabParamList>
>;

export type OrderDetailRouteProp = RouteProp<HomeStackParamList, 'OrderDetail'>;

// Usage in component
import { useNavigation, useRoute } from '@react-navigation/native';

function OrderDetailScreen() {
  const navigation = useNavigation<HomeScreenNavigationProp>();
  const route = useRoute<OrderDetailRouteProp>();

  const { orderId } = route.params; // Type-safe!

  // navigation.navigate() has autocomplete
}
```

---

## 5. State Management (MANDATORY)

### A. Zustand Store

```tsx
// store/useAuthStore.ts
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

interface User {
  id: string;
  email: string;
  name: string;
}

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  // Actions
  setUser: (user: User, token: string) => void;
  logout: () => void;
  setLoading: (loading: boolean) => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: true,

      setUser: (user, token) =>
        set({
          user,
          token,
          isAuthenticated: true,
          isLoading: false,
        }),

      logout: () =>
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        }),

      setLoading: (loading) => set({ isLoading: loading }),
    }),
    {
      name: 'auth-storage',
      storage: createJSONStorage(() => AsyncStorage),
      partialize: (state) => ({
        user: state.user,
        token: state.token,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
);
```

### B. React Query for Server State

```tsx
// features/orders/hooks/useOrders.ts
import { useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { ordersApi } from '../services/ordersApi';

export function useOrders() {
  const {
    data,
    isLoading,
    isRefetching,
    error,
    refetch,
    fetchNextPage,
    hasNextPage,
  } = useInfiniteQuery({
    queryKey: ['orders'],
    queryFn: ({ pageParam = 1 }) => ordersApi.getOrders({ page: pageParam }),
    getNextPageParam: (lastPage) =>
      lastPage.hasMore ? lastPage.page + 1 : undefined,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  const orders = data?.pages.flatMap((page) => page.orders) ?? [];

  return {
    orders,
    isLoading,
    isRefreshing: isRefetching,
    error,
    refetch,
    fetchMore: fetchNextPage,
    hasMore: hasNextPage,
  };
}

export function useCreateOrder() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ordersApi.createOrder,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });
}
```

---

## 6. Styling (MANDATORY)

### A. Theme System

```tsx
// theme/index.ts
export const theme = {
  colors: {
    primary: '#007AFF',
    secondary: '#5856D6',
    success: '#34C759',
    warning: '#FF9500',
    error: '#FF3B30',

    background: {
      primary: '#FFFFFF',
      secondary: '#F2F2F7',
      tertiary: '#E5E5EA',
    },

    text: {
      primary: '#000000',
      secondary: '#3C3C43',
      tertiary: '#8E8E93',
      inverse: '#FFFFFF',
    },

    border: '#C6C6C8',
  },

  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
  },

  borderRadius: {
    sm: 4,
    md: 8,
    lg: 16,
    full: 9999,
  },

  typography: {
    largeTitle: {
      fontSize: 34,
      fontWeight: '700' as const,
      lineHeight: 41,
    },
    title1: {
      fontSize: 28,
      fontWeight: '700' as const,
      lineHeight: 34,
    },
    title2: {
      fontSize: 22,
      fontWeight: '700' as const,
      lineHeight: 28,
    },
    headline: {
      fontSize: 17,
      fontWeight: '600' as const,
      lineHeight: 22,
    },
    body: {
      fontSize: 17,
      fontWeight: '400' as const,
      lineHeight: 22,
    },
    callout: {
      fontSize: 16,
      fontWeight: '400' as const,
      lineHeight: 21,
    },
    caption: {
      fontSize: 12,
      fontWeight: '400' as const,
      lineHeight: 16,
    },
  },
} as const;

export type Theme = typeof theme;
```

### B. Platform-Specific Styles

```tsx
import { Platform, StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  container: {
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
      },
      android: {
        elevation: 4,
      },
    }),
  },
});

// Or use Platform.OS
const hitSlop = Platform.OS === 'ios'
  ? { top: 10, bottom: 10, left: 10, right: 10 }
  : { top: 15, bottom: 15, left: 15, right: 15 };
```

---

## 7. Performance (MANDATORY)

### A. List Optimization

```tsx
import React, { useCallback, useMemo } from 'react';
import { FlatList } from 'react-native';

function OptimizedList({ data }) {
  // Memoize keyExtractor
  const keyExtractor = useCallback((item: Item) => item.id, []);

  // Memoize renderItem
  const renderItem = useCallback(
    ({ item }: { item: Item }) => <ItemCard item={item} />,
    []
  );

  // Memoize getItemLayout for fixed-height items
  const getItemLayout = useCallback(
    (_: any, index: number) => ({
      length: ITEM_HEIGHT,
      offset: ITEM_HEIGHT * index,
      index,
    }),
    []
  );

  return (
    <FlatList
      data={data}
      keyExtractor={keyExtractor}
      renderItem={renderItem}
      getItemLayout={getItemLayout}
      // Performance optimizations
      removeClippedSubviews={true}
      maxToRenderPerBatch={10}
      windowSize={5}
      initialNumToRender={10}
      // Prevent re-renders of unchanged items
      extraData={undefined}
    />
  );
}

// Memoize list item component
const ItemCard = React.memo(function ItemCard({ item }: { item: Item }) {
  return (
    <View style={styles.card}>
      <Text>{item.title}</Text>
    </View>
  );
});
```

### B. Image Optimization

```tsx
import FastImage from 'react-native-fast-image';

function OptimizedImage({ uri, style }) {
  return (
    <FastImage
      source={{
        uri,
        priority: FastImage.priority.normal,
        cache: FastImage.cacheControl.immutable,
      }}
      style={style}
      resizeMode={FastImage.resizeMode.cover}
    />
  );
}
```

### C. Animations with Reanimated

```tsx
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
} from 'react-native-reanimated';

function AnimatedCard({ children }) {
  const scale = useSharedValue(1);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const handlePressIn = () => {
    scale.value = withSpring(0.95);
  };

  const handlePressOut = () => {
    scale.value = withSpring(1);
  };

  return (
    <Pressable onPressIn={handlePressIn} onPressOut={handlePressOut}>
      <Animated.View style={animatedStyle}>{children}</Animated.View>
    </Pressable>
  );
}
```

---

## 8. Native Modules (MANDATORY)

### A. Platform-Specific Code

```tsx
// Using .ios.tsx and .android.tsx files
// components/Haptics.ios.tsx
import * as Haptics from 'expo-haptics';

export function triggerHaptic(type: 'light' | 'medium' | 'heavy') {
  const impact = {
    light: Haptics.ImpactFeedbackStyle.Light,
    medium: Haptics.ImpactFeedbackStyle.Medium,
    heavy: Haptics.ImpactFeedbackStyle.Heavy,
  };
  Haptics.impactAsync(impact[type]);
}

// components/Haptics.android.tsx
import { Vibration } from 'react-native';

export function triggerHaptic(type: 'light' | 'medium' | 'heavy') {
  const duration = { light: 10, medium: 20, heavy: 30 };
  Vibration.vibrate(duration[type]);
}

// Usage (automatically picks correct file)
import { triggerHaptic } from '@/components/Haptics';
```

### B. Native Module Bridge

```tsx
// For custom native functionality
import { NativeModules, Platform } from 'react-native';

const { CustomModule } = NativeModules;

interface CustomModuleInterface {
  processData(data: string): Promise<string>;
  getDeviceInfo(): Promise<DeviceInfo>;
}

export const customModule: CustomModuleInterface = Platform.select({
  ios: CustomModule,
  android: CustomModule,
  default: {
    processData: async () => '',
    getDeviceInfo: async () => ({}),
  },
});
```

---

## 9. Testing (MANDATORY)

### A. Component Tests

```tsx
// __tests__/Button.test.tsx
import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { Button } from '../components/ui/Button';

describe('Button', () => {
  it('renders correctly', () => {
    const { getByText } = render(
      <Button title="Press me" onPress={() => {}} />
    );
    expect(getByText('Press me')).toBeTruthy();
  });

  it('calls onPress when pressed', () => {
    const onPress = jest.fn();
    const { getByText } = render(
      <Button title="Press me" onPress={onPress} />
    );

    fireEvent.press(getByText('Press me'));
    expect(onPress).toHaveBeenCalledTimes(1);
  });

  it('shows loading indicator when loading', () => {
    const { getByTestId, queryByText } = render(
      <Button title="Press me" onPress={() => {}} loading />
    );

    expect(queryByText('Press me')).toBeNull();
    expect(getByTestId('activity-indicator')).toBeTruthy();
  });

  it('is disabled when disabled prop is true', () => {
    const onPress = jest.fn();
    const { getByText } = render(
      <Button title="Press me" onPress={onPress} disabled />
    );

    fireEvent.press(getByText('Press me'));
    expect(onPress).not.toHaveBeenCalled();
  });
});
```

### B. Integration Tests

```tsx
// __tests__/LoginScreen.test.tsx
import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { NavigationContainer } from '@react-navigation/native';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

import { LoginScreen } from '../screens/auth/LoginScreen';
import { authApi } from '../services/authApi';

jest.mock('../services/authApi');

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false } },
});

function renderWithProviders(component: React.ReactElement) {
  return render(
    <QueryClientProvider client={queryClient}>
      <NavigationContainer>{component}</NavigationContainer>
    </QueryClientProvider>
  );
}

describe('LoginScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('submits login form with valid credentials', async () => {
    (authApi.login as jest.Mock).mockResolvedValue({
      user: { id: '1', email: 'test@example.com' },
      token: 'token',
    });

    const { getByPlaceholderText, getByText } = renderWithProviders(
      <LoginScreen />
    );

    fireEvent.changeText(
      getByPlaceholderText('Email'),
      'test@example.com'
    );
    fireEvent.changeText(
      getByPlaceholderText('Password'),
      'password123'
    );
    fireEvent.press(getByText('Login'));

    await waitFor(() => {
      expect(authApi.login).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'password123',
      });
    });
  });

  it('shows error message on login failure', async () => {
    (authApi.login as jest.Mock).mockRejectedValue(
      new Error('Invalid credentials')
    );

    const { getByPlaceholderText, getByText } = renderWithProviders(
      <LoginScreen />
    );

    fireEvent.changeText(getByPlaceholderText('Email'), 'test@example.com');
    fireEvent.changeText(getByPlaceholderText('Password'), 'wrong');
    fireEvent.press(getByText('Login'));

    await waitFor(() => {
      expect(getByText('Invalid credentials')).toBeTruthy();
    });
  });
});
```

---

## 10. Deployment Checklist

### Code Quality
- [ ] TypeScript strict mode
- [ ] No console.log in production
- [ ] All images optimized
- [ ] Unused dependencies removed

### Performance
- [ ] FlatList optimizations applied
- [ ] Memoization where needed
- [ ] No unnecessary re-renders
- [ ] Images lazy loaded

### Platform
- [ ] Tested on iOS and Android
- [ ] Safe area handling
- [ ] Keyboard avoiding behavior
- [ ] Deep linking configured

### Release
- [ ] Version bumped
- [ ] Release notes written
- [ ] App store assets ready
- [ ] Beta tested

---

## 11. Quick Reference

```tsx
// Platform detection
Platform.OS === 'ios'
Platform.select({ ios: value, android: value })

// Safe area
import { useSafeAreaInsets } from 'react-native-safe-area-context';
const insets = useSafeAreaInsets();

// Keyboard
import { KeyboardAvoidingView, Platform } from 'react-native';
<KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'}>

// Dimensions
import { useWindowDimensions } from 'react-native';
const { width, height } = useWindowDimensions();

// Navigation
useNavigation<NavigationType>()
useRoute<RouteType>()
navigation.navigate('Screen', { params })
```

---

**Last Updated:** 2026-01-31
**Version:** 1.0
**Maintainer:** Mobile Team
